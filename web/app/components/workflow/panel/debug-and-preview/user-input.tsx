import {
  memo,
  useState,
} from 'react'
import { useTranslation } from 'react-i18next'
import { ChevronDown } from '@/app/components/base/icons/src/vender/line/arrows'

const UserInput = () => {
  const { t } = useTranslation()
  const [expanded, setExpanded] = useState(true)

  return (
    <div
      className={`
        rounded-xl border
        ${!expanded ? 'bg-indigo-25 border-indigo-100 shadow-none' : 'bg-white shadow-xs border-transparent'}
      `}
    >
      <div
        className={`
          flex items-center px-2 pt-4 h-[18px] text-[13px] font-semibold cursor-pointer
          ${!expanded ? 'text-indigo-800' : 'text-gray-800'}
        `}
        onClick={() => setExpanded(!expanded)}
      >
        <ChevronDown
          className={`mr-1 w-3 h-3 ${!expanded ? '-rotate-90 text-indigo-600' : 'text-gray-300'}`}
        />
        {t('workflow.panel.userInputField').toLocaleUpperCase()}
      </div>
      <div className='px-2 pt-1 pb-3'>
        {
          expanded && (
            <div className='py-2 text-[13px] text-gray-900'>
              <div className='flex px-4 py-1'>
                <div className='shrink-0 mr-4 leading-8'>Service Name</div>
                <input className='grow px-3 h-8 appearance-none outline-none rounded-lg bg-gray-100' />
              </div>
            </div>
          )
        }
      </div>
    </div>
  )
}

export default memo(UserInput)